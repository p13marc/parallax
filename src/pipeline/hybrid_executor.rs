//! Hybrid pipeline executor combining Tokio async tasks with RT threads.
//!
//! **NOTE**: This is legacy code. Use `UnifiedExecutor` instead, which provides
//! automatic execution strategy detection based on `ExecutionHints`.
//!
//! This executor builds on the standard `PipelineExecutor` to support hybrid
//! execution where:
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
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::{Pipeline, HybridExecutor, RtConfig, SchedulingMode};
//!
//! let config = RtConfig {
//!     mode: SchedulingMode::Hybrid,
//!     quantum: 256,
//!     rt_priority: Some(50),
//!     data_threads: 1,
//!     bridge_capacity: 16,
//! };
//!
//! let executor = HybridExecutor::new(config);
//! executor.run(&mut pipeline).await?;
//! ```

use crate::buffer::Buffer;
use crate::element::{AsyncElementDyn, DynAsyncElement, ElementType};
use crate::error::{Error, Result};
use crate::pipeline::rt_bridge::AsyncRtBridge;
use crate::pipeline::rt_scheduler::{
    BoundaryDirection, GraphPartition, RtConfig, RtScheduler, SchedulingMode,
};
use crate::pipeline::{EventReceiver, EventSender, NodeId, Pipeline, PipelineEvent, PipelineState};
use kanal::{AsyncReceiver, AsyncSender, bounded_async};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinHandle;

/// Message passed between pipeline elements.
#[derive(Debug)]
enum Message {
    /// A data buffer.
    Buffer(Buffer),
    /// End of stream signal.
    Eos,
}

/// Handle to a running hybrid pipeline.
pub struct HybridPipelineHandle {
    /// Join handles for Tokio tasks.
    tasks: Vec<JoinHandle<Result<()>>>,
    /// RT thread handles (if any).
    rt_handles: Vec<crate::pipeline::rt_scheduler::DataThreadHandle>,
    /// Event sender for the pipeline.
    events: EventSender,
    /// Bridges at boundaries (kept alive).
    #[allow(dead_code)]
    bridges: Vec<Arc<AsyncRtBridge>>,
}

impl HybridPipelineHandle {
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

        // Signal RT threads to stop and wait for them
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
        // Abort Tokio tasks
        for task in self.tasks {
            task.abort();
        }

        // Signal RT threads to stop
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

/// Hybrid pipeline executor.
///
/// Combines Tokio async tasks for I/O-bound elements with dedicated RT threads
/// for CPU-bound, RT-safe elements.
pub struct HybridExecutor {
    config: RtConfig,
    /// Channel buffer size.
    channel_capacity: usize,
}

impl HybridExecutor {
    /// Create a new hybrid executor with the given configuration.
    pub fn new(config: RtConfig) -> Self {
        Self {
            channel_capacity: config.bridge_capacity,
            config,
        }
    }

    /// Create a hybrid executor with custom channel capacity.
    pub fn with_channel_capacity(mut self, capacity: usize) -> Self {
        self.channel_capacity = capacity;
        self
    }

    /// Get the configuration.
    pub fn config(&self) -> &RtConfig {
        &self.config
    }

    /// Run the pipeline to completion.
    pub async fn run(&self, pipeline: &mut Pipeline) -> Result<()> {
        let handle = self.start(pipeline)?;
        handle.wait().await
    }

    /// Start the pipeline and return a handle.
    ///
    /// This handles state transitions automatically:
    /// - Suspended → Idle (prepare: validate, negotiate)
    /// - Idle → Running (activate)
    pub fn start(&self, pipeline: &mut Pipeline) -> Result<HybridPipelineHandle> {
        // Create event sender
        let events = EventSender::new(256);

        // Transition through states properly
        let old_state = pipeline.state();

        // Prepare if needed (Suspended → Idle)
        if old_state == PipelineState::Suspended {
            pipeline.prepare()?;
            events.send_state_changed(old_state, PipelineState::Idle);
        }

        // Partition the graph based on element affinities
        let mut scheduler = RtScheduler::new(self.config.clone());
        let partition = scheduler.partition_graph(pipeline)?;

        tracing::info!(
            "Graph partition: {} async nodes, {} RT nodes, {} boundary edges",
            partition.async_nodes.len(),
            partition.rt_nodes.len(),
            partition.boundary_edges.len()
        );

        // Determine execution strategy
        let (tasks, rt_handles, bridges) = match self.config.mode {
            SchedulingMode::Async => {
                // Pure async: all nodes in Tokio
                let tasks = self.run_all_async(pipeline, &events)?;
                (tasks, Vec::new(), Vec::new())
            }
            SchedulingMode::Hybrid | SchedulingMode::RealTime => {
                if partition.rt_nodes.is_empty() {
                    // No RT nodes: fall back to pure async
                    tracing::info!("No RT-safe nodes found, running all async");
                    let tasks = self.run_all_async(pipeline, &events)?;
                    (tasks, Vec::new(), Vec::new())
                } else if partition.async_nodes.is_empty() {
                    // All RT: run everything in RT threads
                    tracing::info!("All nodes are RT-safe, running all in RT threads");
                    // LEGACY: Fall back to async - use UnifiedExecutor for proper RT support
                    let tasks = self.run_all_async(pipeline, &events)?;
                    (tasks, Vec::new(), Vec::new())
                } else {
                    // True hybrid: partition between async and RT
                    self.run_hybrid(pipeline, &partition, &mut scheduler, &events)?
                }
            }
        };

        // Activate (Idle → Running)
        let idle_state = pipeline.state();
        pipeline.activate()?;
        events.send_state_changed(idle_state, PipelineState::Running);
        events.send(PipelineEvent::Started);

        Ok(HybridPipelineHandle {
            tasks,
            rt_handles,
            events,
            bridges,
        })
    }

    /// Run all nodes in Tokio async tasks (standard execution).
    fn run_all_async(
        &self,
        pipeline: &mut Pipeline,
        events: &EventSender,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut channels = ChannelNetwork::new();

        // Build channels
        for src_id in pipeline.sources() {
            self.build_channels_from(pipeline, src_id, &mut channels);
        }

        // Spawn tasks
        self.spawn_async_tasks(pipeline, channels, events)
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

        // Set up activation records and dependencies
        scheduler.setup_activations(partition)?;
        scheduler.compute_processing_order(partition, pipeline)?;
        scheduler.setup_dependencies(partition, pipeline)?;

        // Select driver node
        scheduler.select_driver(partition, pipeline);

        // Build channel network for async nodes
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

        // Spawn async tasks (only for async nodes)
        let tasks =
            self.spawn_async_tasks_for_partition(pipeline, partition, channels, scheduler, events)?;

        // Collect bridges
        let mut bridges = Vec::new();
        for edge in &partition.boundary_edges {
            if let Some(bridge) = scheduler.get_bridge(edge.source, edge.sink) {
                bridges.push(bridge);
            }
        }

        // LEGACY: RT thread spawning not implemented in HybridExecutor
        // Use UnifiedExecutor for full RT thread support
        let rt_handles = Vec::new();

        tracing::info!(
            "Hybrid execution: {} async tasks, {} RT threads, {} bridges",
            tasks.len(),
            rt_handles.len(),
            bridges.len()
        );

        Ok((tasks, rt_handles, bridges))
    }

    /// Build channels for async portion of the graph.
    #[allow(clippy::only_used_in_recursion)]
    fn build_channels_for_async(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        async_set: &std::collections::HashSet<NodeId>,
        partition: &GraphPartition,
        scheduler: &RtScheduler,
        network: &mut ChannelNetwork,
    ) {
        let children = pipeline.children(node_id);

        for (child_id, link) in children {
            // Check if this is a boundary edge
            let is_boundary = partition
                .boundary_edges
                .iter()
                .any(|e| e.source == node_id && e.sink == child_id);

            if is_boundary {
                // For boundary edges, we don't create regular channels
                // The bridge handles communication
                continue;
            }

            // Only create channels between async nodes
            if async_set.contains(&child_id) {
                if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                    let (tx, rx) = bounded_async::<Message>(self.channel_capacity);
                    network.add_channel(
                        node_id,
                        link.src_pad.clone(),
                        child_id,
                        link.sink_pad.clone(),
                        tx,
                        rx,
                    );
                }

                // Recurse to children
                self.build_channels_for_async(
                    pipeline, child_id, async_set, partition, scheduler, network,
                );
            }
        }
    }

    /// Build channels recursively (for all-async execution).
    fn build_channels_from(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        network: &mut ChannelNetwork,
    ) {
        let children = pipeline.children(node_id);

        for (child_id, link) in children {
            if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                let (tx, rx) = bounded_async::<Message>(self.channel_capacity);
                network.add_channel(
                    node_id,
                    link.src_pad.clone(),
                    child_id,
                    link.sink_pad.clone(),
                    tx,
                    rx,
                );
            }

            self.build_channels_from(pipeline, child_id, network);
        }
    }

    /// Spawn async tasks for all nodes.
    fn spawn_async_tasks(
        &self,
        pipeline: &mut Pipeline,
        mut channels: ChannelNetwork,
        events: &EventSender,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut tasks = Vec::new();

        // Collect node IDs
        let node_ids: Vec<NodeId> = pipeline
            .sources()
            .into_iter()
            .chain(self.collect_reachable_nodes(pipeline))
            .collect();

        let mut seen = std::collections::HashSet::new();
        let node_ids: Vec<NodeId> = node_ids.into_iter().filter(|id| seen.insert(*id)).collect();

        for node_id in node_ids {
            let task = self.spawn_node_task(pipeline, node_id, &mut channels, events)?;
            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Spawn async tasks only for nodes in the async partition.
    fn spawn_async_tasks_for_partition(
        &self,
        pipeline: &mut Pipeline,
        partition: &GraphPartition,
        mut channels: ChannelNetwork,
        scheduler: &RtScheduler,
        events: &EventSender,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut tasks = Vec::new();

        for &node_id in &partition.async_nodes {
            // Check if this node sends to an RT node (boundary)
            let sends_to_rt: Vec<_> = partition
                .boundary_edges
                .iter()
                .filter(|e| e.source == node_id && e.direction == BoundaryDirection::AsyncToRt)
                .collect();

            // Check if this node receives from an RT node (boundary)
            let receives_from_rt: Vec<_> = partition
                .boundary_edges
                .iter()
                .filter(|e| e.sink == node_id && e.direction == BoundaryDirection::RtToAsync)
                .collect();

            // Get bridges for this node
            let output_bridges: Vec<_> = sends_to_rt
                .iter()
                .filter_map(|e| scheduler.get_bridge(e.source, e.sink))
                .collect();

            let input_bridges: Vec<_> = receives_from_rt
                .iter()
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

    /// Spawn a task for a single node with optional bridges.
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

    /// Collect all reachable nodes from sources.
    fn collect_reachable_nodes(&self, pipeline: &Pipeline) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for src in pipeline.sources() {
            self.collect_reachable_from(pipeline, src, &mut result, &mut visited);
        }

        result
    }

    fn collect_reachable_from(
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
            self.collect_reachable_from(pipeline, child_id, result, visited);
        }
    }
}

impl Default for HybridExecutor {
    fn default() -> Self {
        Self::new(RtConfig::default())
    }
}

// ============================================================================
// Channel Network
// ============================================================================

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
// Task Spawning Functions
// ============================================================================

fn spawn_source_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    outputs: Vec<AsyncSender<Message>>,
    output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("source task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        loop {
            match element.process(None).await {
                Ok(Some(buffer)) => {
                    buffers_processed += 1;

                    // Send to regular channels
                    for tx in &outputs {
                        if tx.send(Message::Buffer(buffer.clone())).await.is_err() {
                            tracing::warn!("source '{}': downstream receiver dropped", name);
                        }
                    }

                    // Send to bridges (async push with backpressure)
                    for bridge in &output_bridges {
                        if bridge.push_async(buffer.clone()).await.is_err() {
                            tracing::warn!("source '{}': bridge push failed", name);
                        }
                    }
                }
                Ok(None) => {
                    tracing::debug!("source '{}' reached EOS", name);
                    for tx in &outputs {
                        let _ = tx.send(Message::Eos).await;
                    }
                    // NOTE: Bridge EOS signaling handled by UnifiedExecutor
                    break;
                }
                Err(e) => {
                    tracing::error!("source '{}' error: {}", name, e);
                    events.send_error(e.to_string(), Some(name.clone()));
                    return Err(e);
                }
            }
        }

        tracing::debug!("source task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
        Ok(())
    })
}

fn spawn_sink_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    input_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("sink task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        // LEGACY: Input bridges not implemented - use UnifiedExecutor
        let _ = input_bridges;

        if let Some(rx) = inputs.into_iter().next() {
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        buffers_processed += 1;
                        if let Err(e) = element.process(Some(buffer)).await {
                            tracing::error!("sink '{}' error: {}", name, e);
                            events.send_error(e.to_string(), Some(name.clone()));
                            return Err(e);
                        }
                    }
                    Ok(Message::Eos) => {
                        tracing::debug!("sink '{}' received EOS", name);
                        break;
                    }
                    Err(_) => {
                        tracing::debug!("sink '{}': channel closed", name);
                        break;
                    }
                }
            }
        }

        tracing::debug!("sink task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
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
        tracing::debug!("transform task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        // LEGACY: Bridges not implemented - use UnifiedExecutor
        let _ = (input_bridges, output_bridges);

        if let Some(rx) = inputs.into_iter().next() {
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        buffers_processed += 1;
                        match element.process(Some(buffer)).await {
                            Ok(Some(out_buffer)) => {
                                for tx in &outputs {
                                    if tx.send(Message::Buffer(out_buffer.clone())).await.is_err() {
                                        tracing::warn!(
                                            "transform '{}': downstream receiver dropped",
                                            name
                                        );
                                    }
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                tracing::error!("transform '{}' error: {}", name, e);
                                events.send_error(e.to_string(), Some(name.clone()));
                                return Err(e);
                            }
                        }
                    }
                    Ok(Message::Eos) => {
                        tracing::debug!("transform '{}' received EOS", name);
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                    Err(_) => {
                        tracing::debug!("transform '{}': channel closed", name);
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                }
            }
        }

        tracing::debug!("transform task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
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
        tracing::debug!("demuxer task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        if let Some(rx) = inputs.into_iter().next() {
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        buffers_processed += 1;
                        match element.process(Some(buffer)).await {
                            Ok(Some(out_buffer)) => {
                                for senders in outputs_by_pad.values() {
                                    for tx in senders {
                                        if tx
                                            .send(Message::Buffer(out_buffer.clone()))
                                            .await
                                            .is_err()
                                        {
                                            tracing::warn!(
                                                "demuxer '{}': downstream receiver dropped",
                                                name
                                            );
                                        }
                                    }
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                tracing::error!("demuxer '{}' error: {}", name, e);
                                events.send_error(e.to_string(), Some(name.clone()));
                                return Err(e);
                            }
                        }
                    }
                    Ok(Message::Eos) => {
                        tracing::debug!("demuxer '{}' received EOS", name);
                        for senders in outputs_by_pad.values() {
                            for tx in senders {
                                let _ = tx.send(Message::Eos).await;
                            }
                        }
                        break;
                    }
                    Err(_) => {
                        tracing::debug!("demuxer '{}': channel closed", name);
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

        tracing::debug!("demuxer task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
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
        tracing::debug!("muxer task '{}' started", name);
        events.send_node_started(&name);

        let mut buffers_processed: u64 = 0;

        let mut receivers: FuturesUnordered<_> = inputs_by_pad
            .into_iter()
            .flat_map(|(pad_name, rxs)| {
                rxs.into_iter().map(move |rx| {
                    let pad = pad_name.clone();
                    async move { (pad, rx.recv().await) }
                })
            })
            .collect();

        let total_inputs = receivers.len();
        let mut eos_count = 0;

        while let Some((pad_name, msg)) = receivers.next().await {
            match msg {
                Ok(Message::Buffer(buffer)) => {
                    buffers_processed += 1;
                    tracing::trace!("muxer '{}' received buffer on pad '{}'", name, pad_name);

                    match element.process(Some(buffer)).await {
                        Ok(Some(out_buffer)) => {
                            for tx in &outputs {
                                if tx.send(Message::Buffer(out_buffer.clone())).await.is_err() {
                                    tracing::warn!("muxer '{}': downstream receiver dropped", name);
                                }
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            tracing::error!("muxer '{}' error: {}", name, e);
                            events.send_error(e.to_string(), Some(name.clone()));
                            return Err(e);
                        }
                    }
                }
                Ok(Message::Eos) => {
                    eos_count += 1;
                    if eos_count >= total_inputs {
                        if let Ok(Some(out_buffer)) = element.process(None).await {
                            for tx in &outputs {
                                let _ = tx.send(Message::Buffer(out_buffer.clone())).await;
                            }
                        }
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                }
                Err(_) => {
                    eos_count += 1;
                    if eos_count >= total_inputs {
                        if let Ok(Some(out_buffer)) = element.process(None).await {
                            for tx in &outputs {
                                let _ = tx.send(Message::Buffer(out_buffer.clone())).await;
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

        tracing::debug!("muxer task '{}' finished", name);
        events.send_node_finished(&name, buffers_processed);
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

    #[tokio::test]
    async fn test_hybrid_executor_creation() {
        let executor = HybridExecutor::new(RtConfig::default());
        assert_eq!(executor.config().mode, SchedulingMode::Async);
    }

    #[tokio::test]
    async fn test_hybrid_executor_async_mode() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 5 })),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
                received: sink_received.clone(),
            })),
        );

        pipeline.link(src, sink).unwrap();

        let executor = HybridExecutor::new(RtConfig::async_only());
        executor.run(&mut pipeline).await.unwrap();

        assert_eq!(sink_received.load(Ordering::Relaxed), 5);
    }

    #[tokio::test]
    async fn test_hybrid_executor_hybrid_mode_no_rt_nodes() {
        // When there are no RT-safe nodes, hybrid mode falls back to async
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 5 })),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
                received: sink_received.clone(),
            })),
        );

        pipeline.link(src, sink).unwrap();

        let executor = HybridExecutor::new(RtConfig::hybrid());
        executor.run(&mut pipeline).await.unwrap();

        assert_eq!(sink_received.load(Ordering::Relaxed), 5);
    }

    #[tokio::test]
    async fn test_graph_partition_all_async() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 5 })),
        );
        let sink_received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
                received: sink_received.clone(),
            })),
        );

        pipeline.link(src, sink).unwrap();

        let scheduler = RtScheduler::new(RtConfig::hybrid());
        let partition = scheduler.partition_graph(&pipeline).unwrap();

        // All nodes should be async (default affinity is Auto, which becomes async for non-RT-safe)
        assert_eq!(partition.async_nodes.len(), 2);
        assert_eq!(partition.rt_nodes.len(), 0);
        assert!(partition.boundary_edges.is_empty());
    }
}
